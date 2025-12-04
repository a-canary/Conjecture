@echo off
echo Setting up Conjecture configuration...
echo.

REM Create config directory
if not exist "%USERPROFILE%\.conjecture" mkdir "%USERPROFILE%\.conjecture"

REM Create default config
echo { > "%USERPROFILE%\.conjecture\config.json"
echo   "providers": [ >> "%USERPROFILE%\.conjecture\config.json"
echo     { >> "%USERPROFILE%\.conjecture\config.json"
echo       "url": "http://localhost:11434", >> "%USERPROFILE%\.conjecture\config.json"
echo       "api": "", >> "%USERPROFILE%\.conjecture\config.json"
echo       "model": "llama2" >> "%USERPROFILE%\.conjecture\config.json"
echo     } >> "%USERPROFILE%\.conjecture\config.json"
echo   ] >> "%USERPROFILE%\.conjecture\config.json"
echo } >> "%USERPROFILE%\.conjecture\config.json"

echo Configuration created at: %USERPROFILE%\.conjecture\config.json
echo.
echo Edit this file to add your LLM providers.
echo.
echo Example providers:
echo   - Local (Ollama): http://localhost:11434
echo   - OpenRouter: https://openrouter.ai/api/v1
echo   - OpenAI: https://api.openai.com/v1
echo.
pause