from collections.abc import Iterator

import pytest

from .integration import RagEnvironment, docker_service, repo_dir, requires_docker_compose

# Text from the demo_umami article about carrots; drupal/scripts/main.sh restricts articles to administrators
RESTRICTED_TEXT = 'purple, red, yellow or white'


@pytest.fixture(scope='module')
def drupal_instance() -> Iterator[str]:
    with docker_service(repo_dir / 'drupal' / 'docker-compose.yml', 'http://localhost:3470') as url:
        yield url


@requires_docker_compose
class TestDrupal():

    def test_drupal_import(self, drupal_instance: str, rag_env: RagEnvironment) -> None:
        loaders = [
            {
                'loader_id': 'drupal_test',
                'loader_type': 'DrupalLoader',
                # Import everything with the consumer from drupal/scripts/create_consumer.php ...
                'auth_type': 'oauth',
                'oauth_import_scope': 'administrator',
                'oauth_client_id': 'test_client',
                'oauth_client_secret': 'test_secret',
                # ... and filter search results by what the chat user may see
                'user_auth_type': 'oauth',
                'base_url': drupal_instance,
                'content_types': ['article', 'recipe'],
            },
        ]

        # Embedding the full recipe texts takes 2-3 minutes on CI runners' CPUs
        rag_env.run_import(loaders, timeout=600)

        # Without the restricted article in the index the leak check below would pass trivially
        assert RESTRICTED_TEXT in rag_env.imported_text(), \
            'restricted article was imported with the administrator scope'

        rag_env.start('pipeline.yml', loaders)

        def restricted_hidden(content: str) -> None:
            assert RESTRICTED_TEXT not in content, 'restricted article must not reach an anonymous user'
        rag_env.ask('Which colors can be carrots?', check=restricted_hidden)

        def public_visible(content: str) -> None:
            assert not content.endswith("Information:\n"), 'contains any document chunks in the prompt'
            # Text from the recipe's ingredients, not its title: recipes have no body field
            assert 'sunflower oil' in content, 'specific text from a test file'
        # Recipes are public, so the authorization filter must keep them
        rag_env.ask('How do I make vegan chocolate and nut brownies?', check=public_visible)
