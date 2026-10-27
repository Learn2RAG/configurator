<?php
declare(strict_types=1);

use Symfony\Component\Validator\Exception\ValidationFailedException;

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
  throw new RuntimeException('Simple OAuth scope entity type not found.');
}

$definition = $definitions[$scope_entity_type_id];
$storage = $entity_type_manager->getStorage($scope_entity_type_id);

$scopes = [
  'authenticated' => [
    'label' => 'Authenticated',
    'description' => 'Access as an authenticated user.',
    'role' => 'authenticated'
  ],
  'administrator' => [
    'label' => 'Administrator',
    'description' => 'Access as an administrator.',
    'role' => 'administrator'
  ],
];

$id_key = $definition->getKey('id') ?: 'id';
$label_key = $definition->getKey('label') ?: 'label';

foreach ($scopes as $id => $data) {
  $scope = $storage->load($id);
  if ($scope === NULL) {
    $scope = $storage->create([
      $id_key => $id,
      $label_key => $data['label'],
      'description' => $data['description'],
      'grant_types' => [
        'authorization_code' => ['status' => TRUE],
        'refresh_token' => ['status' => TRUE],
        'client_credentials' => ['status' => TRUE],
      ],
      'granularity_id' => 'role',
      'granularity_configuration' => [
        'role' => $data['role'],
      ],
    ]);
    $violations = $scope->getTypedData()->validate();
    if ($violations->count() != 0) {
        throw new ValidationFailedException($scope, $violations);
    }
    $scope->save();
    echo "Configured OAuth scope: {$id}\n";
  }
}