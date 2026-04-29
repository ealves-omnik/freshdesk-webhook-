import "dotenv/config";
import express from "express";
import { createClient } from "@supabase/supabase-js";

const log = (...args) => process.stderr.write(args.join(" ") + "\n");

const app = express();
app.use(express.json());

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

// ---------------------------------------------------------------------------
// OpenAI — text-embedding-3-small (1536 dims)
// ---------------------------------------------------------------------------

async function embedTexts(texts) {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({ input: texts, model: "text-embedding-3-small" }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`OpenAI error ${res.status}: ${err}`);
  }

  const data = await res.json();
  return data.data.map((d) => d.embedding);
}

// ---------------------------------------------------------------------------
// Freshdesk
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
    log(`Could not fetch conversations for ticket ${ticketId}: ${res.status}`);
    return [];
  }

  const conversations = await res.json();
  return conversations.map((c) => stripHtml(c.body || "")).filter(Boolean);
}

function stripHtml(html) {
  return html.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
}

// ---------------------------------------------------------------------------
// Chunking
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
// Supabase
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
// Pipeline
// ---------------------------------------------------------------------------

async function processTicket(ticket) {
  const ticketId = ticket.ticket_id || ticket.id;
  const title = ticket.ticket_subject || ticket.subject || `Ticket #${ticketId}`;
  const description = stripHtml(
    ticket.ticket_description || ticket.description_text || ticket.description || ""
  );

  log(`Processing ticket ${ticketId}: ${title}`);

  const conversations = await fetchConversations(ticketId);

  const fullText = [
    `Título: ${title}`,
    description,
    ...conversations.map((c, i) => `Resposta ${i + 1}: ${c}`),
  ]
    .filter(Boolean)
    .join("\n\n");

  if (fullText.length < 20) {
    log(`Ticket ${ticketId} has too little content — skipping.`);
    return 0;
  }

  const textChunks = chunkText(fullText);
  const ticketUrl = `https://${process.env.FRESHDESK_DOMAIN}/helpdesk/tickets/${ticketId}`;

  const embeddings = await embedTexts(textChunks);

  const rows = textChunks.map((content, idx) => ({
    url: ticketUrl,
    title,
    content,
    embedding: embeddings[idx],
    metadata: { source: "freshdesk", ticket_id: ticketId },
  }));

  await insertChunks(rows);
  log(`Ticket ${ticketId} — inserted ${rows.length} chunks.`);
  return rows.length;
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

app.get("/health", (_req, res) => {
  log("health check");
  res.json({ status: "ok" });
});

app.post("/webhook/freshdesk", async (req, res) => {
  log("webhook recebido:", JSON.stringify(req.body));

  const payload = req.body.freshdesk_webhook || req.body;
  const ticket = payload.ticket || payload;

  const status = String(ticket.ticket_status || ticket.status || "").toLowerCase();

  // Freshdesk envia "Resolved" (string) ou 4 (número)
  const isResolved = status === "resolved" || Number(status) === 4;

  if (!isResolved) {
    log(`Ticket ignorado — status: ${status}`);
    return res.json({ ignored: true, status });
  }

  try {
    const count = await processTicket(ticket);
    res.json({ ok: true, chunks: count });
  } catch (err) {
    log("Erro:", err.message);
    res.status(500).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => log(`Freshdesk webhook listening on :${PORT}`));
