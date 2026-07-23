<?php
declare(strict_types=1);

use Drupal\Core\Entity\EntityStorageException;
use Symfony\Component\Validator\Exception\ValidationFailedException;

$scope_id = 'authenticated';
$scope_label = 'Authenticated';
$scope_description = 'Access as an authenticated Drupal user.';

$entity_type_manager = \Drupal::entityTypeManager();
$definitions = $entity_type_manager->getDefinitions();

$scope_entity_type_id = NULL;
foreach ($definitions as $entity_type_id => $definition) {
  $provider = (string) $definition->getProvider();

  if (
    str_contains($entity_type_id, 'scope')
    && str_contains($provider, 'simple_oauth')
    && $definition->getGroup() === 'configuration'
  ) {
    $scope_entity_type_id = $entity_type_id;
    break;
  }
}
if ($scope_entity_type_id === NULL) {
  throw new RuntimeException('Simple OAuth scope entity type not found. Check if simple_oauth_static_scope is installed and enabled.');
}

$definition = $definitions[$scope_entity_type_id];
$storage = $entity_type_manager->getStorage($scope_entity_type_id);
$scope = $storage->load($scope_id);
if ($scope !== NULL) {
  throw new RuntimeException(sprintf('OAuth scope %s already exists', $scope_id));
}

$id_key = $definition->getKey('id') ?: 'id';
$label_key = $definition->getKey('label') ?: 'label';
$scope = $storage->create([
  $id_key => $scope_id,
  $label_key => $scope_label,
  'description' => $scope_description,
  'grant_types' => [
    'authorization_code' => ['status' => TRUE],
    'refresh_token' => ['status' => TRUE],
  ],
]);

$violations = $scope->getTypedData()->validate();
if ($violations->count() != 0) {
    throw new ValidationFailedException($scope, $violations);
}
$scope->save();
