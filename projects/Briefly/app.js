const SYSTEM_INSTRUCTIONS = `Act as a professional plain-language editor. Deconstruct dense jargon into simple terms or analogies. Format the output into three distinct sections: 1. The Bottom Line (2-sentence summary), 2. Key Terms (glossary of complex words used), and 3. The Breakdown (the full simplified summary with clean headings).

Return the response in Markdown and use the section titles exactly as written.`;

const GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models";
const STORAGE_KEY = "briefly.geminiApiKey";
const MODEL_KEY = "briefly.model";
const PREFS_KEY = "briefly.preferences";

const SAMPLE_ARTICLE = `Whereas the contemporary regulatory environment increasingly obligates organizations to demonstrate proactive data governance, covered entities must implement commensurate technical and organizational safeguards, document the rationale for retention schedules, and periodically reassess whether collected information remains necessary for the originally specified processing purpose.

In practice, this requires cross-functional collaboration among legal, security, product, and operations teams to establish auditable workflows that minimize unnecessary data collection while preserving sufficient operational continuity for customer support, compliance reporting, and legitimate business analytics.`;

const AUDIENCE_OPTIONS = {
  general: "a general professional audience",
  executive: "busy executives who need decisions and risks quickly",
  student: "students or new readers who need concepts explained from the ground up",
};

const DETAIL_OPTIONS = {
  brief: "Keep the output tight and prioritize only the most important ideas.",
  balanced: "Balance brevity with enough context to make the article useful.",
  detailed: "Include more supporting context while keeping the language plain.",
};

const articleInput = document.querySelector("#articleInput");
const apiKeyInput = document.querySelector("#apiKeyInput");
const modelInput = document.querySelector("#modelInput");
const rememberKeyInput = document.querySelector("#rememberKeyInput");
const detailSelect = document.querySelector("#detailSelect");
const analogiesInput = document.querySelector("#analogiesInput");
const simplifyButton = document.querySelector("#simplifyButton");
const sampleButton = document.querySelector("#sampleButton");
const clearButton = document.querySelector("#clearButton");
const copyButton = document.querySelector("#copyButton");
const downloadButton = document.querySelector("#downloadButton");
const settingsButton = document.querySelector("#settingsButton");
const settingsDialog = document.querySelector("#settingsDialog");
const toggleKeyButton = document.querySelector("#toggleKeyButton");
const removeKeyButton = document.querySelector("#removeKeyButton");
const outputSurface = document.querySelector("#outputSurface");
const inputMeta = document.querySelector("#inputMeta");
const outputMeta = document.querySelector("#outputMeta");
const statusText = document.querySelector("#statusText");
const keyStatus = document.querySelector("#keyStatus");
const audienceInputs = [...document.querySelectorAll('input[name="audience"]')];

let latestOutput = "";
let latestOriginal = "";

function init() {
  const savedKey = getStorageValue(STORAGE_KEY);
  const savedModel = getStorageValue(MODEL_KEY);
  const savedPrefs = getSavedPreferences();

  if (savedKey) {
    apiKeyInput.value = savedKey;
    rememberKeyInput.checked = true;
  }

  if (savedModel) {
    modelInput.value = savedModel;
  }

  applyPreferences(savedPrefs);

  articleInput.addEventListener("input", updateInputMeta);
  apiKeyInput.addEventListener("input", handleCredentialChange);
  modelInput.addEventListener("input", handleModelChange);
  rememberKeyInput.addEventListener("change", handleCredentialChange);
  detailSelect.addEventListener("change", savePreferences);
  analogiesInput.addEventListener("change", savePreferences);
  audienceInputs.forEach((input) => input.addEventListener("change", savePreferences));
  simplifyButton.addEventListener("click", simplifyArticle);
  sampleButton.addEventListener("click", loadSampleArticle);
  clearButton.addEventListener("click", clearArticle);
  copyButton.addEventListener("click", copyOutput);
  downloadButton.addEventListener("click", downloadOutput);
  outputSurface.addEventListener("click", handleOutputAction);
  settingsButton.addEventListener("click", openSettings);
  toggleKeyButton.addEventListener("click", toggleApiKeyVisibility);
  removeKeyButton.addEventListener("click", removeSavedKey);

  settingsDialog.addEventListener("click", (event) => {
    if (event.target === settingsDialog) {
      settingsDialog.close();
    }
  });

  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      simplifyArticle();
    }
  });

  updateInputMeta();
  updateKeyStatus();
  updateOutputActions();
  refreshIcons();
}

function getStorageValue(key) {
  try {
    return localStorage.getItem(key) || "";
  } catch {
    return "";
  }
}

function setStorageValue(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Browsers can block storage in private modes.
  }
}

function removeStorageValue(key) {
  try {
    localStorage.removeItem(key);
  } catch {
    // Browsers can block storage in private modes.
  }
}

function getSavedPreferences() {
  try {
    return JSON.parse(getStorageValue(PREFS_KEY)) || {};
  } catch {
    return {};
  }
}

function applyPreferences(preferences) {
  const audience = preferences.audience && AUDIENCE_OPTIONS[preferences.audience] ? preferences.audience : "general";
  const audienceInput = audienceInputs.find((input) => input.value === audience);
  if (audienceInput) {
    audienceInput.checked = true;
  }

  if (preferences.detail && DETAIL_OPTIONS[preferences.detail]) {
    detailSelect.value = preferences.detail;
  }

  if (typeof preferences.analogies === "boolean") {
    analogiesInput.checked = preferences.analogies;
  }
}

function savePreferences() {
  setStorageValue(
    PREFS_KEY,
    JSON.stringify({
      audience: getSelectedAudience(),
      detail: detailSelect.value,
      analogies: analogiesInput.checked,
    }),
  );
}

function refreshIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

function updateInputMeta() {
  const stats = getTextStats(articleInput.value);
  inputMeta.textContent = stats.words
    ? `${formatNumber(stats.words)} words | ${stats.minutes} min read`
    : `${formatNumber(stats.characters)} characters`;
}

function getTextStats(text) {
  const trimmed = text.trim();
  const words = trimmed ? trimmed.split(/\s+/).filter(Boolean).length : 0;
  const characters = trimmed.length;
  return {
    characters,
    words,
    minutes: Math.max(1, Math.ceil(words / 220)),
  };
}

function formatNumber(value) {
  return value.toLocaleString();
}

function handleCredentialChange() {
  const key = apiKeyInput.value.trim();

  if (rememberKeyInput.checked && key) {
    setStorageValue(STORAGE_KEY, key);
  } else {
    removeStorageValue(STORAGE_KEY);
  }

  updateKeyStatus();
}

function updateKeyStatus() {
  const hasKey = Boolean(apiKeyInput.value.trim());
  keyStatus.classList.toggle("is-ready", hasKey);
  keyStatus.querySelector("span").textContent = hasKey ? "API key ready" : "API key needed";
}

function handleModelChange() {
  const model = normalizeModelName(modelInput.value);
  if (model) {
    setStorageValue(MODEL_KEY, model);
  }
}

function normalizeModelName(model) {
  return model.trim().replace(/^models\//i, "");
}

function openSettings() {
  if (typeof settingsDialog.showModal === "function") {
    settingsDialog.showModal();
    apiKeyInput.focus();
    return;
  }

  settingsDialog.setAttribute("open", "");
  apiKeyInput.focus();
}

function toggleApiKeyVisibility() {
  const isHidden = apiKeyInput.type === "password";
  apiKeyInput.type = isHidden ? "text" : "password";
  toggleKeyButton.setAttribute("aria-label", isHidden ? "Hide API key" : "Show API key");
  toggleKeyButton.innerHTML = isHidden ? '<i data-lucide="eye-off"></i>' : '<i data-lucide="eye"></i>';
  refreshIcons();
}

function removeSavedKey() {
  apiKeyInput.value = "";
  rememberKeyInput.checked = false;
  removeStorageValue(STORAGE_KEY);
  updateKeyStatus();
  apiKeyInput.focus();
}

function loadSampleArticle() {
  articleInput.value = SAMPLE_ARTICLE;
  articleInput.focus();
  updateInputMeta();
  setStatus("Sample loaded.");
}

function clearArticle() {
  articleInput.value = "";
  articleInput.focus();
  updateInputMeta();
  setStatus("");
}

async function copyOutput() {
  if (!latestOutput) return;

  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(latestOutput);
    } else {
      copyWithFallback(latestOutput);
    }

    outputMeta.textContent = "Copied";
    window.setTimeout(() => {
      outputMeta.textContent = getOutputMeta(latestOriginal, latestOutput);
    }, 1400);
  } catch {
    outputMeta.textContent = "Copy failed";
  }
}

function copyWithFallback(text) {
  const helper = document.createElement("textarea");
  helper.value = text;
  helper.setAttribute("readonly", "");
  helper.style.position = "fixed";
  helper.style.left = "-999px";
  document.body.appendChild(helper);
  helper.select();
  document.execCommand("copy");
  helper.remove();
}

function downloadOutput() {
  if (!latestOutput) return;

  const blob = new Blob([latestOutput], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `briefly-${new Date().toISOString().slice(0, 10)}.md`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function handleOutputAction(event) {
  const actionButton = event.target.closest("[data-action]");
  if (!actionButton) return;

  if (actionButton.dataset.action === "open-settings") {
    openSettings();
  }

  if (actionButton.dataset.action === "retry") {
    simplifyArticle();
  }
}

async function simplifyArticle() {
  const article = articleInput.value.trim();
  const apiKey = apiKeyInput.value.trim();
  const model = normalizeModelName(modelInput.value);

  if (!article) {
    setStatus("Paste an article first.");
    articleInput.focus();
    return;
  }

  if (!apiKey) {
    setStatus("Add your Gemini API key in settings.");
    openSettings();
    return;
  }

  if (!model) {
    setStatus("Add a Gemini model name in settings.");
    openSettings();
    modelInput.focus();
    return;
  }

  setLoading(true);
  renderLoadingState();
  setStatus("Simplifying...");
  outputMeta.textContent = "Processing";

  try {
    const preferences = getCurrentPreferences();
    const result = await requestSimplification({ article, apiKey, model, preferences });
    latestOriginal = article;
    latestOutput = result;
    renderResult(article, result);
    setStatus("Done.");
    refreshIcons();
  } catch (error) {
    latestOutput = "";
    updateOutputActions();
    outputMeta.textContent = "Needs attention";
    renderError(error.message || "Something went wrong.");
    setStatus("Unable to simplify.");
  } finally {
    setLoading(false);
  }
}

function getCurrentPreferences() {
  return {
    audience: getSelectedAudience(),
    detail: detailSelect.value,
    analogies: analogiesInput.checked,
  };
}

function getSelectedAudience() {
  return audienceInputs.find((input) => input.checked)?.value || "general";
}

function buildUserPrompt(article, preferences) {
  const audience = AUDIENCE_OPTIONS[preferences.audience] || AUDIENCE_OPTIONS.general;
  const detail = DETAIL_OPTIONS[preferences.detail] || DETAIL_OPTIONS.balanced;
  const analogyInstruction = preferences.analogies
    ? "Use simple analogies when they genuinely make a complex idea easier to understand."
    : "Avoid analogies unless they are essential for clarity.";

  return `Simplify this article for ${audience}.

Preferences:
- ${detail}
- ${analogyInstruction}
- Preserve important caveats, numbers, names, and risks.
- Do not invent facts that are not in the source article.

Article:
${article}`;
}

async function requestSimplification({ article, apiKey, model, preferences }) {
  const response = await fetch(`${GEMINI_ENDPOINT}/${encodeURIComponent(model)}:generateContent`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": apiKey,
    },
    body: JSON.stringify({
      system_instruction: {
        parts: [{ text: SYSTEM_INSTRUCTIONS }],
      },
      contents: [
        {
          role: "user",
          parts: [
            {
              text: buildUserPrompt(article, preferences),
            },
          ],
        },
      ],
      generationConfig: {
        temperature: 0.35,
        topP: 0.9,
        maxOutputTokens: 4096,
      },
    }),
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(getGeminiError(payload, response.status));
  }

  const text = payload?.candidates?.[0]?.content?.parts
    ?.map((part) => part.text || "")
    .join("\n")
    .trim();

  if (!text) {
    const blockReason = payload?.promptFeedback?.blockReason;
    throw new Error(blockReason ? `Gemini blocked the request: ${blockReason}.` : "Gemini returned an empty response.");
  }

  return text;
}

function getGeminiError(payload, status) {
  const message = payload?.error?.message;
  if (message) {
    return message;
  }

  return `Gemini request failed with status ${status}.`;
}

function renderLoadingState() {
  outputSurface.classList.add("is-busy");
  outputSurface.innerHTML = `
    <div class="loading-state" aria-label="Simplifying article">
      <div class="loading-card">
        <div class="loading-title"></div>
        <div class="skeleton-lines">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
      <div class="loading-card">
        <div class="loading-title"></div>
        <div class="skeleton-lines">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>
  `;
}

function renderResult(original, result) {
  outputSurface.classList.remove("is-busy");
  outputSurface.innerHTML = `
    ${renderResultStats(original, result)}
    <article class="result-content">${renderMarkdown(result)}</article>
  `;
  outputMeta.textContent = getOutputMeta(original, result);
  updateOutputActions();
}

function renderResultStats(original, result) {
  const originalStats = getTextStats(original);
  const resultStats = getTextStats(result);
  const change = getReductionLabel(originalStats.words, resultStats.words);

  return `
    <div class="result-meta-strip" aria-label="Simplification statistics">
      <div class="result-stat">
        <span>Original</span>
        <strong>${formatNumber(originalStats.words)} words</strong>
      </div>
      <div class="result-stat">
        <span>Simplified</span>
        <strong>${formatNumber(resultStats.words)} words</strong>
      </div>
      <div class="result-stat">
        <span>Change</span>
        <strong>${escapeHtml(change)}</strong>
      </div>
    </div>
  `;
}

function getOutputMeta(original, result) {
  if (!result) return "Ready when you are";

  const originalWords = getTextStats(original).words;
  const outputWords = getTextStats(result).words;
  return `${formatNumber(outputWords)} words | ${getReductionLabel(originalWords, outputWords)}`;
}

function getReductionLabel(originalWords, outputWords) {
  if (!originalWords || !outputWords) {
    return "No comparison";
  }

  const reduction = Math.round((1 - outputWords / originalWords) * 100);
  if (reduction > 0) {
    return `${reduction}% shorter`;
  }

  if (reduction < 0) {
    return `${Math.abs(reduction)}% expanded`;
  }

  return "same length";
}

function renderError(message) {
  outputSurface.classList.remove("is-busy");
  outputSurface.innerHTML = `
    <div class="error-box">
      <strong>Could not simplify the article.</strong>
      <p>${escapeHtml(message)}</p>
      <div class="error-actions">
        <button class="ghost-button" type="button" data-action="open-settings">
          <i data-lucide="settings"></i>
          <span>Settings</span>
        </button>
        <button class="primary-button small" type="button" data-action="retry">
          <i data-lucide="rotate-cw"></i>
          <span>Retry</span>
        </button>
      </div>
    </div>
  `;
  refreshIcons();
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let listType = "";

  const closeList = () => {
    if (listType) {
      html.push(`</${listType}>`);
      listType = "";
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (!line) {
      closeList();
      continue;
    }

    const sectionMatch = line.match(/^(?:#{1,3}\s*)?(?:\d+\.\s*)?(The Bottom Line|Key Terms|The Breakdown):?\s*$/i);
    if (sectionMatch) {
      closeList();
      html.push(`<h2>${escapeHtml(sectionMatch[1])}</h2>`);
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      closeList();
      const level = Math.min(3, Math.max(2, headingMatch[1].length + 1));
      html.push(`<h${level}>${formatInline(headingMatch[2])}</h${level}>`);
      continue;
    }

    const unorderedMatch = line.match(/^[-*]\s+(.+)$/);
    if (unorderedMatch) {
      if (listType !== "ul") {
        closeList();
        html.push("<ul>");
        listType = "ul";
      }
      html.push(`<li>${formatInline(unorderedMatch[1])}</li>`);
      continue;
    }

    const orderedMatch = line.match(/^\d+\.\s+(.+)$/);
    if (orderedMatch) {
      if (listType !== "ol") {
        closeList();
        html.push("<ol>");
        listType = "ol";
      }
      html.push(`<li>${formatInline(orderedMatch[1])}</li>`);
      continue;
    }

    closeList();
    html.push(`<p>${formatInline(line)}</p>`);
  }

  closeList();
  return html.join("");
}

function formatInline(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setLoading(isLoading) {
  simplifyButton.disabled = isLoading;
  sampleButton.disabled = isLoading;
  clearButton.disabled = isLoading;
  simplifyButton.classList.toggle("is-loading", isLoading);
  articleInput.disabled = isLoading;
  detailSelect.disabled = isLoading;
  analogiesInput.disabled = isLoading;
  audienceInputs.forEach((input) => {
    input.disabled = isLoading;
  });
}

function updateOutputActions() {
  const hasOutput = Boolean(latestOutput);
  copyButton.disabled = !hasOutput;
  downloadButton.disabled = !hasOutput;
}

function setStatus(message) {
  statusText.textContent = message;
}

init();
