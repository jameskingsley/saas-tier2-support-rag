# OAuth2 & Bearer Token Authentication

All API requests must include a valid Bearer token in the Authorization header. Expired tokens return HTTP 401 Unauthorized; refresh tokens every 24 hours.