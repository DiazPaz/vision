// ===============================
// Control de 2 motores DC
// Arduino Nano + Puente H
// ===============================

// Motor A
#define IN1 3   // PWM
#define IN2 5   // PWM

// Motor B
#define IN3 6   // PWM
#define IN4 9   // PWM

// Velocidades (0–255)
int speedA = 150;
int speedB = 150;

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  stopMotors();
}

void loop() {

  // Motores hacia adelante
  moveForward(speedA, speedB);
  delay(2000);

  // Alto
  stopMotors();
  delay(1000);

  // Motores hacia atrás
  moveBackward(speedA, speedB);
  delay(2000);

  // Giro a la izquierda
  turnLeft(120);
  delay(1500);

  // Giro a la derecha
  turnRight(120);
  delay(1500);

  stopMotors();
  delay(3000);
}

// ===============================
// Funciones de movimiento
// ===============================

void moveForward(int spA, int spB) {
  analogWrite(IN1, spA);
  analogWrite(IN2, 0);

  analogWrite(IN3, spB);
  analogWrite(IN4, 0);
}

void moveBackward(int spA, int spB) {
  analogWrite(IN1, 0);
  analogWrite(IN2, spA);

  analogWrite(IN3, 0);
  analogWrite(IN4, spB);
}

void turnLeft(int sp) {
  // Motor A atrás, Motor B adelante
  analogWrite(IN1, 0);
  analogWrite(IN2, sp);
 
  analogWrite(IN3, sp);
  analogWrite(IN4, 0);
}

void turnRight(int sp) {
  // Motor A adelante, Motor B atrás
  analogWrite(IN1, sp);
  analogWrite(IN2, 0);

  analogWrite(IN3, 0);
  analogWrite(IN4, sp);
}

void stopMotors() {
  analogWrite(IN1, 0);
  analogWrite(IN2, 0);
  analogWrite(IN3, 0);
  analogWrite(IN4, 0);
}
