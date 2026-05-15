#include <Wire.h>
#include <Octoliner.h>

Octoliner octoliner;

//motor_pins
#define M1A 27   // right 
#define M1B 26
#define M2A 25   // left
#define M2B 33

// pwm
#define PWM_LEFT  0
#define PWM_RIGHT 1


int threshold = 3950;   // adjust after Serial Monitor test

float Kp = 0.24;
float Kd = 0.105;

int baseSpeed = 160;
int prevError = 0;

// sensor weights (0 = RIGHT, 7 = LEFT)
int weights[8] = {-350, -250, -150, -50, 50, 150, 250, 350};


void setup() {
  Serial.begin(115200);

  octoliner.begin();
  octoliner.setSensitivity(200);

  pinMode(M1B, OUTPUT);
  pinMode(M2B, OUTPUT);

  
  ledcAttach(M2A, 5000, 8);   // left motor
  ledcAttach(M1A, 5000, 8);   // right motor
}


void setMotor(int leftSpeed, int rightSpeed) {

  leftSpeed = constrain(leftSpeed, 0, 255);
  rightSpeed = constrain(rightSpeed, 0, 255);

  ledcWrite(M2A, leftSpeed);
  digitalWrite(M2B, LOW);

  ledcWrite(M1A, rightSpeed);
  digitalWrite(M1B, LOW);
}


void loop() {

  int sum = 0;
  int active = 0;

  for (int i = 0; i < 8; i++) {
    int val = octoliner.analogRead(i);

    Serial.print(val);
    Serial.print("\t");

    if (val > threshold) {
      sum += weights[i];
      active++;
    }
  }

  Serial.println();

  int error;

  if (active == 0) {
    error = prevError;
  } else {
    error = sum / active;
  }

  int derivative = error - prevError;

  float correction = Kp * error + Kd * derivative;

  // smooth motor control
  int leftSpeed  = baseSpeed + correction;
  int rightSpeed = baseSpeed - correction;

  // safety limit
  leftSpeed  = constrain(leftSpeed, 0, 255);
  rightSpeed = constrain(rightSpeed, 0, 255);

  setMotor(leftSpeed, rightSpeed);

  prevError = error;

  delay(5);
}