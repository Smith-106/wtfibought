# alpha 因子挖掘：FastAPI + LangGraph，行情 parquet 与 sqlite 都落 /data；管理前端由它自己托管在 /alpha/

# 前端：在 node 里构建，只把 dist 带进运行镜像
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.13-slim

# uv 官方镜像里的单文件二进制
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# 装完就预编译 pyc；硬链接跨层不通，改成拷贝
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 先只拷清单装依赖，改 src 时这层还能命中缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev

# main.py 按 src 往上两级找 web/dist
COPY --from=web /web/dist web/dist

EXPOSE 8084

# 退出时最多等 5 秒：交易员页的实时推送不会自己结束，超时就取消请求、照常关库（docker stop 10 秒后强杀）
CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "wiib_alpha.main:app", "--host", "0.0.0.0", "--port", "8084", \
     "--timeout-graceful-shutdown", "5"]
