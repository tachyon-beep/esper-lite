# Secret Management Guidance

Esper-Lite uses a shared signing key to protect Leyline/Oona payloads. This
document outlines generation, storage, verification, and rotation procedures.

## Generation

Generate a random 256-bit secret and distribute it via a secure channel (e.g.,
vault, password manager). Example command:

```bash
python - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
```

Store the value in the `ESPER_LEYLINE_SECRET` environment variable. Do **not**
commit the secret to version control. The repo ships with `.env.example` so you
can copy and populate local `.env` files.

## Runtime Configuration

- `OonaClient` automatically loads the secret from `ESPER_LEYLINE_SECRET`. If
  the variable is absent, signing is skipped and payloads are accepted without
  verification (intended only for local development).
- To enforce signing in tests or local runs set the variable explicitly:

  ```bash
  export ESPER_LEYLINE_SECRET="<hex-secret>"
  pytest tests/oona/test_messaging.py
  ```

- When multiple services run in separate processes, ensure each process has the
  same secret in its environment.

## Verification

- Signatures are added as base64-encoded `X-Leyline-Signature` fields on Redis
  stream entries.
- Consumers verify payloads before invoking handlers. Tampered payloads are
  skipped; producers should monitor for increases in dropped acknowledgements.

## Rotation Procedure

1. Generate a new secret and distribute it securely.
2. Update producers and consumers to accept both old and new secrets during the
   transition window if zero downtime is required. The helper can be extended to
   accept multiple secrets; currently, redeploy with the new secret in lockstep.
3. After all services use the new secret, revoke the old secret from vault.

## Storage

- For production or shared environments, store the secret in a secret manager
  (e.g., HashiCorp Vault, AWS Secrets Manager) and inject it via environment
  variables at runtime.
- For local development, use a `.env` file (gitignored) or export the variable
  manually before running the stack.

## Future Work

- Rotate secrets automatically via a secret management service.
- Extend signing to other Leyline transports (gRPC, HTTP) as they are added.

