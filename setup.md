# Quick Setup
Download Poetry here

## Steps
```zsh
poetry init
poetry add <packages>
poetry install --no-root
poetry env info # get's a venv location
```
### VScode
1. Ctrl+P: `Python Select Interpreter`
1. Copy the env location found from `poetry env info` into it.
1. Enter

### CLI
1. Windows Please do `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process` 
1. in order to run the `poetry env activate` command
