@echo off
rem Update local Docker Compose service with rebuilt image and run smoke test
setlocal enabledelayedexpansion

set SERVICE=chatterbox-api

























exit /b 0
necho ==== All done ====)    exit /b 1    echo [ERROR] Smoke test failed or returned non-zero exit codeif errorlevel 1 (
n:: Use docker compose exec with environment variable; -T avoids tty issues
ndocker compose exec -T -e PYTHONPATH=src %SERVICE% python3 tests/smoke_test_speed.py
necho ==== Running smoke test inside container ====
necho (This runs `tests/smoke_test_speed.py` using PYTHONPATH=src)docker compose logs --no-color --tail=200 %SERVICE%
necho ==== Recent logs (tail 200) ====timeout /t 5 /nobreak >nul
necho Waiting 5 seconds for container to initialize...)    exit /b 1    echo [ERROR] docker compose up failedif errorlevel 1 (docker compose up -d --force-recreate --remove-orphans %SERVICE%
necho ==== Bringing up service ====)    exit /b 1    echo [ERROR] docker compose build failedif errorlevel 1 (docker compose build --no-cache %SERVICE%necho ==== Rebuilding image for %SERVICE% ====