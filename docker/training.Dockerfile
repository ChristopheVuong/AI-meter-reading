# Stage 1: Build dependencies
FROM python:3.9-slim AS builder

# Install Poetry (use pipx for isolation)
RUN pip install pipx && pipx install poetry

# Set environment variables for Poetry
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN git clone https://github.com/ChristopheVuong/AI-meter-reading /AI-meter-reading

# Copy only dependency files to leverage Docker caching
WORKDIR /AI-meter-reading
COPY pyproject.toml poetry.lock ./

# Install dependencies (including dev dependencies if needed)
RUN poetry install --no-root --no-interaction --no-ansi

# Stage 2: Final image
FROM python:3.9-slim

# Copy dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy application code
WORKDIR /AI-meter-reading
COPY . .

# Run the script via Poetry
# Run chained Poetry scripts
CMD ["sh", "-c", "poetry run download-model-paddle && poetry run train-model-paddle --with-logging --log-level=info"]