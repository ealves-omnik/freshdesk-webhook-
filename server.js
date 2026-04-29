import "dotenv/config";
import express from "express";
import { createClient } from "@supabase/supabase-js";

const app = express();
app.use(express.json());

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

// ---------------------------------------------------------------------------
// Voyage AI — embed a list of texts (1024 dims, same model as the Python RAG)
// ---------------------------------------------------------------------------

async function embedTexts(texts) {
  const res = await fetch("https://api.voyageai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.VOYAGE_API_KEY}`,
    },
    body: JSON.stringify({ input: texts, model: "voyage-3", input_type: "document" }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Voyage AI error ${res.status}: ${err}`);
  }

  const data = await res.json();
  return data.data.map((d) => d.embedding);
}

// ---------------------------------------------------------------------------
// Freshdesk — fetch all conversation messages for a ticket
// ---------------------------------------------------------------------------

async function fetchConversations(ticketId) {
  const url = `https://${process.env.FRESHDESK_DOMAIN}/api/v2/tickets/${ticketId}/conversations`;
  const res = await fetch(url, {
    headers: {
      Authorization:
        "Basic " +
        Buffer.from(`${process.env.FRESHDESK_API_KEY}:X`).toString("base64"),
    },
  });

  if (!res.ok) {
    console.warn(`Could not fetch conversations for ticket ${ticketId}: ${res.status}`);
    return [];
  }

  const conversations = await res.json();
  return conversations.map((c) => stripHtml(c.body || "")).filter(Boolean);
}

function stripHtml(html) {
  return html.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
}

// ---------------------------------------------------------------------------
// Chunking — split long text into overlapping pieces (same params as Python)
// ---------------------------------------------------------------------------

function chunkText(text, chunkSize = 900, overlap = 150) {
  const chunks = [];
  let start = 0;

  while (start < text.length) {
    let end = start + chunkSize;

    if (end < text.length) {
      const boundary = text.lastIndexOf(". ", end);
      if (boundary > start + chunkSize / 2) end = boundary + 1;
    }

    const chunk = text.slice(start, end).trim();
    if (chunk.length > 60) chunks.push(chunk);

    start = end - overlap;
  }

  return chunks;
}

// ---------------------------------------------------------------------------
// Insert chunks into Supabase (same table as the Python indexer)
// ---------------------------------------------------------------------------

async function insertChunks(chunks) {
  const BATCH = 50;

  for (let i = 0; i < chunks.length; i += BATCH) {
    const batch = chunks.slice(i, i + BATCH);
    const { error } = await supabase.from("documents").insert(batch);
    if (error) throw new Error(`Supabase insert error: ${error.message}`);
  }
}

// ---------------------------------------------------------------------------
// Main pipeline: ticket → text → chunks → embeddings → Supabase
// ---------------------------------------------------------------------------

async function processTicket(ticket) {
  const ticketId = ticket.id;
  const title = ticket.subject || `Ticket #${ticketId}`;
  const description = stripHtml(ticket.description_text || ticket.description || "");

  const conversations = await fetchConversations(ticketId);

  // Build a single text block: title + description + all replies
  const fullText = [
    `Título: ${title}`,
    description,
    ...conversations.map((c, i) => `Resposta ${i + 1}: ${c}`),
  ]
    .filter(Boolean)
    .join("\n\n");

  if (fullText.length < 100) {
    console.log(`Ticket ${ticketId} has too little content — skipping.`);
    return 0;
  }

  const textChunks = chunkText(fullText);
  const ticketUrl = `https://${process.env.FRESHDESK_DOMAIN}/helpdesk/tickets/${ticketId}`;

  // Embed all chunks (Voyage AI free tier: stay under 10K TPM)
  const embeddings = await embedTexts(textChunks);

  const rows = textChunks.map((content, idx) => ({
    url: ticketUrl,
    title,
    content,
    embedding: embeddings[idx],
    metadata: { source: "freshdesk", ticket_id: ticketId },
  }));

  await insertChunks(rows);
  console.log(`Ticket ${ticketId} — inserted ${rows.length} chunks.`);
  return rows.length;
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

app.get("/health", (_req, res) => res.json({ status: "ok" }));

app.post("/webhook/freshdesk", async (req, res) => {
  // Freshdesk sends the ticket inside a "freshdesk_webhook" wrapper
  const payload = req.body.freshdesk_webhook || req.body;
  const ticket = payload.ticket || payload;

  const status = ticket.ticket_status ?? ticket.status;
  const RESOLVED = 4; // Freshdesk status code for "Resolved"

  if (Number(status) !== RESOLVED) {
    // Ignore tickets that aren't resolved
    return res.json({ ignored: true, status });
  }

  try {
    const count = await processTicket(ticket);
    res.json({ ok: true, chunks: count });
  } catch (err) {
    console.error("Error processing ticket:", err);
    res.status(500).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Freshdesk webhook listening on :${PORT}`));
