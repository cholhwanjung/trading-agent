# trading-agent — gateway/scheduler 공용 이미지 (개인 사용 → 배포 경로 동일)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy TZ=Asia/Seoul

# 의존성 레이어 캐시 (소스 변경과 분리)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY adapters/ adapters/
COPY alpha_lab/ alpha_lab/
COPY eval/ eval/
COPY harness/ harness/
COPY interaction/ interaction/
COPY llm/ llm/
COPY memory/ memory/
COPY reflection/ reflection/
COPY risk/ risk/
COPY trader/ trader/
COPY scripts/ scripts/
RUN uv sync --frozen --no-dev

# 상태·로그는 볼륨 (compose 가 ./data 마운트)
VOLUME /app/data

CMD ["uv", "run", "uvicorn", "interaction.api:app", "--host", "0.0.0.0", "--port", "8721"]
