<?php
declare(strict_types=1);

use Drupal\user\Entity\User;

$username = 'normal_user';
$password = 'user';
$email = 'normal_user@example.com';

$storage = \Drupal::entityTypeManager()->getStorage('user');
if (!$storage->loadByProperties(['name' => $username])) {
  $user = User::create([
    'name' => $username,
    'mail' => $email,
    'status' => 1,
    'pass' => $password,
  ]);

  $user->save();
  echo "User created: {$username}\n";
}