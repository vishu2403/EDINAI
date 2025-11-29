/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // INAI Dark Theme Colors
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
        gray: {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6b7280',
          600: '#4b5563',
          700: '#374151',
          800: '#1f2937',
          900: '#111827',
          950: '#0a0f1a',
        },
        dark: {
          bg: '#0a0f1a',
          card: '#1a1f2e',
          border: '#2d3748',
          text: {
            primary: '#ffffff',
            secondary: '#a0aec0',
            muted: '#718096',
          }
        },
        accent: {
          blue: '#4299e1',
          green: '#48bb78',
          yellow: '#ed8936',
          red: '#f56565',
          purple: '#9f7aea',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'scale-in': 'scaleIn 0.2s ease-out',
        'bounce-gentle': 'bounceGentle 2s infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        slideDown: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        scaleIn: {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        bounceGentle: {
          '0%, 100%': { transform: 'translateY(-5px)' },
          '50%': { transform: 'translateY(0)' },
        },
      },
      boxShadow: {
        'neumorphic': '20px 20px 60px #0a0f1a, -20px -20px 60px #1a1f2e',
        'neumorphic-inset': 'inset 20px 20px 60px #0a0f1a, inset -20px -20px 60px #1a1f2e',
        'glow': '0 0 20px rgba(66, 153, 225, 0.3)',
        'glow-intense': '0 0 40px rgba(66, 153, 225, 0.5)',
      },
      backdropBlur: {
        xs: '2px',
      },
      borderRadius: {
        'xl': '1rem',
        '2xl': '1.5rem',
        '3xl': '2rem',
      },
    },
  },
  plugins: [],
}
