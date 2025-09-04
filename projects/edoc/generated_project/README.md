# SimpleCalculator

## Description
SimpleCalculator is a lightweight, web‑based calculator that performs basic arithmetic operations. It provides an intuitive button interface, supports keyboard input, and offers responsive design for desktop and mobile devices.

## Tech Stack
- **HTML** – structure of the calculator UI
- **CSS** – styling and responsive layout (media queries)
- **JavaScript** – core functionality, event handling, and calculation logic

## Features
- Basic arithmetic: addition, subtraction, multiplication, division
- Clear entry (`C`) and all clear (`AC`) functions
- Keyboard shortcuts for numbers, operators, `Enter` (equals), `Backspace` (delete), and `Esc` (clear)
- Real‑time error handling (e.g., division by zero displays an error message)
- Responsive design that adapts to various screen sizes via CSS media queries

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/simplecalculator.git
   cd simplecalculator
   ```
2. Open `index.html` in any modern web browser (no build step or server required).

## Usage Guide
- **Button Layout**: The calculator displays a numeric keypad (0‑9), a decimal point, arithmetic operators (`+`, `-`, `*`, `/`), an equals (`=`) button, and clear functions (`C` for clear entry, `AC` for all clear).
- **Keyboard Shortcuts**:
  - Numbers `0`‑`9` and `.` input the corresponding characters.
  - `+`, `-`, `*`, `/` perform the respective operations.
  - `Enter` or `=` computes the result.
  - `Backspace` deletes the last character.
  - `Esc` clears the current entry (`C`).
- **Error Handling**: Invalid operations (e.g., division by zero) display an error message on the screen and reset the calculator state.

## Responsive Design Note
The UI uses CSS media queries to adjust button sizes and layout for smaller screens, ensuring a comfortable touch experience on mobile devices.

## Contribution Guidelines
1. Fork the repository and create a new branch for your feature or bug fix.
2. Follow the existing code style (ES6 syntax, descriptive variable names, and modular functions).
3. To extend functionality (e.g., adding scientific functions like `sin`, `cos`, `log`),
   - Update `script.js` with the new operation logic.
   - Add corresponding buttons in `index.html`.
   - Style them in `style.css`/`styles.css`.
4. Submit a pull request with a clear description of the changes.

## License
MIT License (placeholder) – see `LICENSE` file for details.
