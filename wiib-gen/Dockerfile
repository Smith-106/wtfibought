# gen 指令生成器：FastAPI + LangGraph，checkpoint 落 /data 的 sqlite
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

EXPOSE 8083

CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "wiib_gen.main:app", "--host", "0.0.0.0", "--port", "8083"]
