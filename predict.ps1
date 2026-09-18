param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Comando
)

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
$exe = Join-Path $root ".venv\Scripts\predictor.exe"

if (-not (Test-Path $exe)) {
    Write-Host "No encontre el predictor en .venv. Configuralo con:" -ForegroundColor Red
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host '  .\.venv\Scripts\python.exe -m pip install -e ".[dev]"' -ForegroundColor Yellow
    exit 1
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker no esta disponible. Inicialo o agregalo al PATH." -ForegroundColor Red
    exit 1
}

Write-Host "Levantando Postgres..." -ForegroundColor Cyan
docker compose up -d postgres | Out-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "No se pudo levantar Postgres." -ForegroundColor Red
    exit $LASTEXITCODE
}

$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    docker compose exec -T postgres pg_isready | Out-Null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $ready) {
    Write-Host "Postgres no respondio a tiempo." -ForegroundColor Red
    exit 1
}

$env:POSTGRES_HOST = "localhost"

if (-not $Comando -or $Comando.Count -eq 0) {
    & $exe --help
} else {
    & $exe @Comando
}
exit $LASTEXITCODE