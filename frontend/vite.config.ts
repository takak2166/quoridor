import { defineConfig } from "vite";

import { readPortEnv } from "./envPorts";

const backendPort = readPortEnv("BACKEND_PORT", "8000");
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
