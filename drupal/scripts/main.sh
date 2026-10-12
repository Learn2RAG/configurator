#!/bin/sh
set -eux
composer require --no-audit --no-blocking --no-progress --with-all-dependencies 'drupal/simple_oauth:^6.1' 'drush/drush:*'
php -d memory_limit=256M web/core/scripts/drupal install --password=test --no-interaction demo_umami

# Assign admin role explicitly
drush urol administrator admin

drush config:set system.logging error_level verbose
# simple_oauth_static_scope: provides 'user' granularity_id
drush pm:install -v jsonapi simple_oauth simple_oauth_static_scope consumers

drush config:set simple_oauth.settings access_token_expiration 31536000 -y

drush cr

scripts/configure_oauth_keys.sh

# Generate the custom access module
mkdir -p web/modules/custom/api_access

cat <<'EOF' > web/modules/custom/api_access/api_access.info.yml
name: API Access Control
type: module
description: Restricts view access to articles for non-admins.
core_version_requirement: ^10 || ^11
package: Custom
EOF

cat <<'EOF' > web/modules/custom/api_access/api_access.module
<?php
use Drupal\Core\Access\AccessResult;
use Drupal\node\NodeInterface;
use Drupal\Core\Session\AccountInterface;
use Drupal\Core\Database\Query\AlterableInterface;

function api_access_node_access(NodeInterface $node, $op, AccountInterface $account) {
  if ($op === 'view' && $node->bundle() === 'article') {
    if (!$account->hasPermission('bypass node access') && !$account->hasPermission('administer nodes') && !in_array('administrator', $account->getRoles())) {
      return AccessResult::forbidden()->cachePerPermissions()->addCacheableDependency($node);
    }
  }
  return AccessResult::neutral();
}

function api_access_query_node_access_alter(AlterableInterface $query) {
  $account = \Drupal::currentUser();

  // Allow access if the user has bypass permission OR if they are authenticated via an OAuth token with the administrator role
  if (!$account->hasPermission('bypass node access') && !$account->hasPermission('administer nodes') && !in_array('administrator', $account->getRoles())) {
    foreach ($query->getTables() as $alias => $table_info) {
      if (isset($table_info['table']) && $table_info['table'] === 'node_field_data') {
        $query->condition($alias . '.type', 'article', '<>');
        break;
      }
    }
  }
}
EOF

# Enable the module
drush en api_access -y

drush scr scripts/configure_oauth_scope.php

# CRITICAL FIX 2: Flush cache again so the newly created config scopes are visible to the consumer validation
drush cr

drush scr scripts/create_consumer.php
drush scr scripts/create_normal_user.php

drush cr

(cd web && # https://www.drupal.org/project/drupal/issues/3150146
# add --host=0.0.0.0 for remote access
php -d memory_limit=256M core/scripts/drupal server --host=0.0.0.0 --port=80 --suppress-login --no-interaction
)