#define LED_PIN 13
void setup() {
    // DUmmy arduino code
    Serial.begin(9600);
}

void loop() {





  
      Serial.print("ax:");Serial.print(random(300)); Serial.print(",");
      Serial.print("ay:");Serial.print(random(300)); Serial.print(",");
      Serial.print("az:");Serial.print(random(300)); Serial.print(",");
      Serial.print("gx:");Serial.print(random(300)); Serial.print(",");
      Serial.print("gy:");Serial.print(random(300)); Serial.print(",");
      Serial.print("gz:");Serial.println(random(300));
      delay(10);
}
