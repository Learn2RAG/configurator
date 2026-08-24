#!/bin/sh
set -eux
composer require --with-all-dependencies 'drupal/simple_oauth:^6.1' 'drush/drush:*'
php -d memory_limit=256M web/core/scripts/drupal install --password=test --no-interaction demo_umami
drush config:set system.logging error_level verbose
# simple_oauth_static_scope: provides 'user' granularity_id
drush pm:install -v jsonapi simple_oauth simple_oauth_static_scope
scripts/configure_oauth_keys.sh
drush scr scripts/configure_oauth_scope.php
drush scr scripts/create_consumer.php

(cd web && # https://www.drupal.org/project/drupal/issues/3150146
# add --host=0.0.0.0 for remote access
php -d memory_limit=256M core/scripts/drupal server --host=0.0.0.0 --port=80 --suppress-login --no-interaction
)
