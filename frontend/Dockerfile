FROM node:20-slim

WORKDIR /app

# Copy the lockfile too and use `npm ci` so the image gets the exact dependency
# versions that were committed, rather than whatever resolves at build time.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev"]
