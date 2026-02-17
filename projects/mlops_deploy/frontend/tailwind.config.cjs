/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef6ff",
          100: "#d9e9ff",
          200: "#b3d3ff",
          300: "#8cbcff",
          400: "#66a6ff",
          500: "#408fff",
          600: "#1a79ff",
          700: "#005fe6",
          800: "#004ab3",
          900: "#003480"
        }
      }
    }
  },
  plugins: []
};
