import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// 产物由 wiib-alpha 自己托管在 /alpha/
export default defineConfig({
  base: '/alpha/',
  plugins: [react(), tailwindcss()],
  server: {
    port: 3002,
    proxy: {
      '/api/alpha': 'http://localhost:8084',
    },
  },
})
