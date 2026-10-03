# Frontend React/Vite — build statyczny serwowany przez nginx.
# Etap 1: budowa aplikacji (Node) -> katalog dist/
FROM node:22-alpine AS build

WORKDIR /app

# Najpierw manifesty zależności (lepsze cache warstw Dockera).
# --legacy-peer-deps: @tailwindcss/vite deklaruje peer vite <=7, projekt używa vite 8.
COPY package.json package-lock.json ./
RUN npm ci --legacy-peer-deps || npm install --legacy-peer-deps --no-audit --no-fund

# Reszta źródeł i produkcyjny build (tsc -b && vite build).
COPY . .
RUN npm run build

# Etap 2: serwowanie statycznego buildu przez nginx z proxy do API.
FROM nginx:1.27-alpine

# Konfiguracja serwera (SPA + proxy /predict, /explain, /chat, /agent ... -> api:8000).
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Artefakty buildu z etapu 1.
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
