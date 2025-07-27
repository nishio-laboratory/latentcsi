import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/control': 'http://0.0.0.0:8123',
      '/ws': {
        target: 'ws://0.0.0.0:8123',
        ws: true
      }
    }
  }
})
