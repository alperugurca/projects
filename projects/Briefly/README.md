# Briefly - Article Simplifier

Briefly is a lightweight, client-side web application designed to act as a professional plain-language editor. It deconstructs dense jargon, reports, policy memos, or research excerpts into simple terms and analogies using the Google Gemini API.

## Features

- **Audience Targeting**: Tailor the simplification for a general professional audience, busy executives, or students.
- **Adjustable Depth**: Choose between brief, balanced, or detailed summaries based on your needs.
- **Smart Formatting**: Automatically structures output into three clear sections:
  1. The Bottom Line (A concise 2-sentence summary)
  2. Key Terms (A glossary of complex words used)
  3. The Breakdown (The full simplified summary with clean headings)
- **Analogy Toggles**: Option to include or exclude simple analogies to make complex ideas easier to grasp.
- **Privacy-First**: Operates entirely in the browser using your own Gemini API key. Data is communicated directly to the Gemini API securely.

## Getting Started

1. Clone or download the repository to your local machine.
2. Open `index.html` in any modern web browser.
3. Click the **Settings** (gear) icon in the top right corner.
4. Provide your **Gemini API Key** and preferred model (e.g., `gemini-2.5-flash`).
5. Paste any dense article or use the provided sample text, and click **Simplify Article**.

## Tech Stack

- HTML5
- CSS3 (Vanilla)
- Vanilla JavaScript
- [Lucide Icons](https://lucide.dev/)
- [Gemini API](https://ai.google.dev/)

## License

This project is provided for educational and demonstrative purposes.
