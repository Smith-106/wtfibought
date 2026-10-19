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
      // 因子挖掘接口去 wiib-alpha，其余 /api（登录）去 sim；前缀长的先匹配
      '/api/alpha': 'http://localhost:8084',
      '/api': 'http://localhost:8080',
    },
  },
})
