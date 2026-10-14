import { defineConfig } from "vite";

const backendPort = process.env.BACKEND_PORT ?? "8000";
const backendOrigin = `http://127.0.0.1:${backendPort}`;

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      "/api": backendOrigin,
      "/health": backendOrigin,
    },
  },
});
