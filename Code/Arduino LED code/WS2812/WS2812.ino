#include <Adafruit_NeoPixel.h>

#define LED_PIN     6
#define NUMPIXELS   24
#define BRIGHTNESS  30

Adafruit_NeoPixel strip(NUMPIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);

bool ledsOn = false;

void setup() {
  Serial.begin(9600); // Start serial communication
  strip.begin();
  strip.setBrightness(BRIGHTNESS);
  strip.clear();
  strip.show();

  Serial.println("Type 'on' to turn LEDs on, 'off' to turn them off.");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim(); 

    if (command == "on") {
      ledsOn = true;

      for (int i = 1; i < NUMPIXELS; i += 2) {
        strip.setPixelColor(i, strip.Color(210, 245, 125));
      }

      strip.show();
      Serial.println("LEDs turned ON");
    } else if (command == "off") {
      ledsOn = false;
      strip.clear();
      strip.show();
      Serial.println("LEDs turned OFF");
    } else {
      Serial.println("Unknown command. Type 'on' or 'off'.");
    }
  }
}



