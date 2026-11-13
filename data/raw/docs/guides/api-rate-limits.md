# API Rate Limiting & Quotas

Standard tier accounts are restricted to 100 requests per minute. Exceeding this limit returns an HTTP 429 Too Many Requests response header with a Retry-After timestamp.