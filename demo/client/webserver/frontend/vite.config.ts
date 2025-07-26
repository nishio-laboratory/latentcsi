import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/control': 'http://localhost:8123',
      '/ws': {
        target: 'ws://localhost:8123',
        ws: true
      }
    }
  }
})
