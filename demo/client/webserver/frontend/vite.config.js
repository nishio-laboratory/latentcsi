import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/control': 'http://localhost:8123',
      '/ws': {
        target: 'ws://localhost:8123',
        ws: true,
      },
    },
  },
})
