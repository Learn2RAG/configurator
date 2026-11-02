from typing import Any

from ..userui.auth.oauth import discover_applications


def _discover(**loader: Any) -> dict[str, Any]:
    config = {'loaders': [{
        'loader_id': 'drupal',
        'user_auth_type': 'oauth',
        'oauth_client_id': 'test_client',
        'oauth_client_secret': 'test_secret',
        'base_url': 'http://localhost:3470/',
        **loader,
    }]}
    return next(discover_applications(config))[2]


def test_empty_urls_from_the_form_use_defaults() -> None:
    oauth_kwargs = _discover(oauth_authorize_url='', oauth_access_token_url='')
    assert oauth_kwargs['authorize_url'] == 'http://localhost:3470/oauth/authorize'
    assert oauth_kwargs['access_token_url'] == 'http://localhost:3470/oauth/token'


def test_configured_urls_are_kept() -> None:
    oauth_kwargs = _discover(oauth_authorize_url='https://sso.example.com/auth', oauth_access_token_url='https://sso.example.com/token')
    assert oauth_kwargs['authorize_url'] == 'https://sso.example.com/auth'
    assert oauth_kwargs['access_token_url'] == 'https://sso.example.com/token'


def test_default_scope_is_least_privilege() -> None:
    assert _discover()['client_kwargs'] == {'scope': 'authenticated'}
    assert _discover(oauth_scope='administrator')['client_kwargs'] == {'scope': 'administrator'}
