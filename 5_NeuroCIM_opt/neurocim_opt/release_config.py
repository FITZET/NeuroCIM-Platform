"""Public release endpoints embedded in packaged executables.

The OAuth client ID is public by design. The matching client secret and every
administrator token must stay in GitHub Actions secrets and are never packaged.
"""

GITHUB_OAUTH_CLIENT_ID = ""
GITHUB_OWNER = "Lowpower-Lab"
GITHUB_DISTRIBUTION_REPOSITORY = "NeuroCIM-Distribution"
ACCESS_PORTAL_URL = "https://fitzet.github.io/NeuroCIM-Access/"

