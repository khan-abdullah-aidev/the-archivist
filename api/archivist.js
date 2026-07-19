// Server-side OpenRouter proxy for The Archivist.
// Keeps the OpenRouter API key out of client code — set OPENROUTER_API_KEY
// as an environment variable in the Vercel project settings.

const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';
const MODEL = 'nvidia/nemotron-3-ultra-550b-a55b:free';
const REQUEST_TIMEOUT_MS = 45000;

const MAX_DRAFT_CHARS = 20000;
const MAX_IMPORT_CHARS = 12000;
const MAX_LORE_CONTEXT_CHARS = 20000;

// Extracts the first balanced {...} object from the model's raw text, tolerating
// leading/trailing commentary and reasoning-tag variants the model may leak
// (Nemotron/DeepSeek/Qwen don't all use the same tag name, and "exclude" reasoning
// isn't guaranteed to fully suppress it).
function extractJsonObject(raw) {
  let text = raw.replace(/```json|```/gi, '');
  text = text.replace(/<(think|thinking|reasoning|scratchpad)>[\s\S]*?<\/\1>/gi, '').trim();

  try {
    return JSON.parse(text);
  } catch {
    // fall through to brace-matching below
  }

  const start = text.indexOf('{');
  if (start === -1) throw new Error('No JSON object found in model output.');

  let depth = 0;
  for (let i = start; i < text.length; i++) {
    if (text[i] === '{') depth++;
    else if (text[i] === '}') {
      depth--;
      if (depth === 0) {
        return JSON.parse(text.slice(start, i + 1));
      }
    }
  }
  throw new Error('Unterminated JSON object in model output.');
}

async function callOpenRouter(apiKey, messages, maxTokens, signal) {
  const upstream = await fetch(OPENROUTER_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
      'HTTP-Referer': 'https://the-archivist.app',
      'X-Title': 'The Archivist',
    },
    body: JSON.stringify({
      model: MODEL,
      messages,
      max_tokens: maxTokens,
      reasoning: { exclude: true },
    }),
    signal,
  });
  return upstream;
}

export default async function handler(req, res) {
  // Never let a failed/rate-limited call get cached as if it were a valid response.
  res.setHeader('Cache-Control', 'no-store');

  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed.' });
    return;
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: 'Server is not configured with an OpenRouter API key.' });
    return;
  }

  const body = req.body || {};
  const { action } = body;

  let messages;
  let maxTokens;

  if (action === 'scan') {
    const { systemPrompt, userPrompt } = body;
    if (typeof systemPrompt !== 'string' || typeof userPrompt !== 'string' || userPrompt.length < 30) {
      res.status(400).json({ error: 'Invalid scan request.' });
      return;
    }
    if (userPrompt.length > MAX_LORE_CONTEXT_CHARS + MAX_DRAFT_CHARS) {
      res.status(400).json({ error: 'Draft + lore context is too large.' });
      return;
    }
    messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ];
    maxTokens = 4000;
  } else if (action === 'import') {
    const { prompt } = body;
    if (typeof prompt !== 'string' || prompt.length < 100) {
      res.status(400).json({ error: 'Invalid import request.' });
      return;
    }
    if (prompt.length > MAX_IMPORT_CHARS + 2000) {
      res.status(400).json({ error: 'Import text is too large.' });
      return;
    }
    messages = [{ role: 'user', content: prompt }];
    maxTokens = 6000;
  } else {
    res.status(400).json({ error: 'Unknown or missing action.' });
    return;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const upstream = await callOpenRouter(apiKey, messages, maxTokens, controller.signal);
    clearTimeout(timeout);

    if (upstream.status === 429) {
      res.status(429).json({ error: 'OpenRouter is rate-limiting requests right now. Try again shortly.' });
      return;
    }
    if (!upstream.ok) {
      const errText = await upstream.text().catch(() => '');
      res.status(502).json({ error: `OpenRouter error (${upstream.status}): ${errText.slice(0, 300) || 'unknown error'}` });
      return;
    }

    const data = await upstream.json();
    if (data.error) {
      res.status(502).json({ error: data.error.message || 'OpenRouter returned an error.' });
      return;
    }
    const choice = data.choices?.[0];
    if (!choice || choice.error) {
      res.status(502).json({ error: choice?.error?.message || 'Provider dropped the connection. Try again.' });
      return;
    }
    const raw = choice.message?.content;
    if (!raw) {
      res.status(502).json({ error: 'The model returned an empty response. Try again.' });
      return;
    }

    let parsed;
    try {
      parsed = extractJsonObject(raw);
    } catch (e) {
      const snippet = raw.slice(0, 300).replace(/\s+/g, ' ').trim();
      res.status(502).json({ error: `The model returned malformed JSON. Raw output started with: "${snippet}"` });
      return;
    }

    res.status(200).json({ result: parsed });
  } catch (err) {
    clearTimeout(timeout);
    if (err.name === 'AbortError') {
      res.status(504).json({ error: 'The request to OpenRouter timed out. Try again.' });
      return;
    }
    res.status(500).json({ error: err.message || 'Unexpected server error.' });
  }
}
