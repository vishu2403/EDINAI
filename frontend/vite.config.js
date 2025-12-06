import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3020,
    host: true,
    proxy: {
      '/api': {
        // target: 'http://13.234.223.188:8000',
        target: 'http://localhost:1171',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/admin-api': {
        // target: 'http://13.234.223.188:8000',
        target: 'http://localhost:1171',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/admin-api/, ''),
      },
      '/uploads': {
        // target: 'http://13.234.223.188:8000',
        target: 'http://localhost:1171',
        changeOrigin: true,
      },
      '/chapter-materials': {
        // target: 'http://13.234.223.188:8000',
        target: 'http://localhost:1171',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          ui: ['lucide-react', 'framer-motion', 'react-hot-toast'],
        },
      },
    },
  },
})
