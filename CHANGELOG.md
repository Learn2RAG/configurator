# Changelog
## [Unreleased]
## [1.0.0] - 2026-09-16
### Added
- MediaWiki data import
- Jira data import
- chat history awareness
- per-document user access control (Drupal)
- pipeline auto-optimization
- retrieval optimization
- displaying status during import

### Fixed
- offline installation
- Windows process management

### Changed
- open webui chat UI replaced with llama.cpp's UI
- interface improvements
- performance improvements

### Removed
- unused API endpoints were removed

## [0.3.0] - 2026-06-02
### Added
- continuous import with updating the indexed data
- Sharepoint import and simple configuration interface
- Drupal import and configuration interface
- ability to clean pipeline's storage without removing the pipeline
- simple status messages for the import process
- OpenAI-compatible API for RAG
### Fixed
- uninstall script
- interface language selection when browser accepts both
- web data source configuration
- missing import logs in the terminal
### Changed
- by default configuration interface and chat listen on localhost only
- in config.yml "host" and "port" options are moved under "UI"; see also "CHAT"
- openwebui upgraded to a newer version
### Removed
- openwebui-pipelines (not needed anymore)
