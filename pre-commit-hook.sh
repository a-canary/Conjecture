#!/bin/sh
# Pre-commit hook to detect potential API keys and sensitive data
# Install: cp pre-commit-hook.sh .git/hooks/pre-commit

echo "🔍 Running security checks..."

# Check for potential API keys in staged files
API_KEY_PATTERNS="AIza|sk-[a-zA-Z0-9]{20,}|gsk_[a-zA-Z0-9]{20,}|api[_-]?key[\s=:]+['\"][a-zA-Z0-9_-]{20,}['\"]"

if git diff --cached --name-only | xargs grep -E "$API_KEY_PATTERNS" 2>/dev/null; then
    echo ""
    echo "❌ SECURITY ALERT: Potential API key detected in staged files!"
    echo ""
    echo "Please:"
    echo "1. Remove the API key from the code"
    echo "2. Use environment variables instead"
    echo "3. Add the pattern to .gitignore if it's a false positive"
    echo ""
    echo "Example of secure usage:"
    echo "  api_key = os.getenv('API_KEY')"
    echo ""
    exit 1
fi

# Check for other sensitive patterns
SENSITIVE_PATTERNS="password[\s=:]+['\"][^'\"]{8,}['\"]|secret[\s=:]+['\"][^'\"]{8,}['\"]|token[\s=:]+['\"][^'\"]{8,}['\"]"

if git diff --cached --name-only | xargs grep -E "$SENSITIVE_PATTERNS" 2>/dev/null; then
    echo ""
    echo "⚠️  WARNING: Potential sensitive data detected in staged files!"
    echo ""
    echo "Please review and ensure no passwords, secrets, or tokens are hardcoded."
    echo "Use environment variables for sensitive configuration."
    echo ""
    exit 1
fi

echo "✅ Security checks passed!"
exit 0