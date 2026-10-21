/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#f5f5f1",
          900: "#ffffff",
          850: "#eef0eb",
          800: "#e6e9e3",
          750: "#d9ded7",
          text: "#20322d",
          dim: "#66756f",
          faint: "#89958f",
        },
        parchment: "#D9D2BE",
        forensic: "#1f7562",
        risk: {
          high: "#B54B42",
          medium: "#B98530",
          low: "#4B7A63",
        },
      },
      fontFamily: {
        serif: ["Lora", "ui-serif", "Georgia", "serif"],
        sans: ["Inter", "ui-sans-serif", "system-ui"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
