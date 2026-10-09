<?php
declare(strict_types=1);

use Drupal\consumers\Entity\Consumer;
use Symfony\Component\Validator\Exception\ValidationFailedException;

$label = 'Test consumer';
$secret = 'test_secret';
$ownerId = 1;
$redirectUrl = 'http://localhost:9001/auth/oauth/callback';
$grantTypes = ['authorization_code', 'refresh_token', 'client_credentials'];
// We include both roles here so the consumer has access to request them
$scopeIds = ['authenticated', 'administrator'];

$setFieldValues = static function (
  \Drupal\Core\Entity\ContentEntityInterface $entity,
  string $fieldName,
  array $values,
): void {
  if (!$entity->hasField($fieldName)) {
    return;
  }
  $definition = $entity->get($fieldName)->getFieldDefinition()->getFieldStorageDefinition();
  $mainProperty = $definition->getMainPropertyName();
  if (!$mainProperty) {
    throw new RuntimeException(sprintf('No main property for field: %s', $fieldName));
  }
  $entity->set($fieldName, array_map(
    static fn(string $value): array => [$mainProperty => $value],
    $values,
  ));
};

$storage = \Drupal::entityTypeManager()->getStorage('consumer');

$existingIds = $storage->getQuery()->accessCheck(FALSE)->condition('client_id', 'test_client')->execute();
if ($existingIds) {
  $storage->delete($storage->loadMultiple($existingIds));
}

$consumer = Consumer::create([
  'label' => $label,
  'description' => 'Test consumer.',
  'client_id' => 'test_client',
  'user_id' => $ownerId,
  'secret' => $secret,
  'confidential' => TRUE,
  'third_party' => FALSE,
  'roles' => $scopeIds,
]);

$setFieldValues($consumer, 'redirect', [$redirectUrl]);
$setFieldValues($consumer, 'grant_types', $grantTypes);

$violations = $consumer->validate();
if ($violations->count() != 0) {
    throw new ValidationFailedException($consumer, $violations);
}
$consumer->save();