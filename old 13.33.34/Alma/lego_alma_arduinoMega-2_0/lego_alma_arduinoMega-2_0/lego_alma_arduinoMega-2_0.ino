// ---------------------------------------------------------------------------------------------------------
// ALMA Lego Microcontroller for Arduino Mega
//
//
// Sven Karlsson, 2023-04-13
//
// Based on "ALMA Lego Microcontroller" by Felix Stoehr
//
// This Arduino board reads the contacts of the antennas and of the additional buttons (hour-angle, sources) and returns a string with the
// closed contacts.
//
// The data are sent over the USB bus only when there is a state change. The microcontroller checks the state of the contacts every 50ms,
// allowing for a very rapid reaction and at the same time reducing the bounces from the contacts.
//
// Each contact should be connected between gpio pin and GND.
//
// Reading with python
// -------------------
//
// pip3 install pyserial
//
// import serial
// ser = serial.Serial('/dev/ttyACM0')
// while True:
//    print(ser.readline().decode('utf-8'))
//
// ---------------------------------------------------------------------------------------------------------

byte startPin = 3;
byte pins = 45;

byte endPin = startPin + pins;
int key;
String result;
String lastresult = "";
String angleState = "100";
String modeState = "1000";
static const uint8_t buttons[] = { A0, A1, A2, A3, A4, A5, A6 };
static const uint8_t buttonsLEDS[] = { A7, A8, A9, A10, A11, A12, A13 };


void setup() {
  // Reed Switch Inputs
  for (int x = startPin; x < endPin; x++) {
    pinMode(x, INPUT_PULLUP);
  }
  // Buttons Input
  for (int x = 0; x <= 6; x++) {
    pinMode(buttons[x], INPUT_PULLUP);
  }
  // Buttons LED output
  for (int x = 0; x <= 6; x++) {
    pinMode(buttonsLEDS[x], OUTPUT);
  }
  digitalWrite(buttonsLEDS[0], HIGH);
  digitalWrite(buttonsLEDS[3], HIGH);

  Serial.begin(115200);
}


void loop() {
  int angleButton[3];
  int modeButton[4];
  result = "";

  // Read Reedswitches
  for (int pin = startPin; pin < endPin; pin++) {

    // invert value so that closed is 1 and open is 0
    key = !digitalRead(pin);

    // construct the output string
    result = result + String(key);
  }

  // Read and set angle buttons
  for (int i = 0; i <= 2; i++) {
    angleButton[i] = !digitalRead(buttons[i]);
  }

  if (angleButton[0] + angleButton[1] + angleButton[2] == 1) {
    angleState = String(angleButton[0]) + String(angleButton[1]) + String(angleButton[2]);
    for (int y = 0; y <= 2; y++) {
      digitalWrite(buttonsLEDS[y], angleButton[y]);
    }
  }
  result = result + angleState;

  // Read and set mode buttons
  for (int i = 0; i <= 3; i++) {
    modeButton[i] = !digitalRead(buttons[i+3]);
  }

  if (modeButton[0] + modeButton[1] + modeButton[2] + modeButton[3] == 1) {
    modeState = String(modeButton[0]) + String(modeButton[1]) + String(modeButton[2]) + String(modeButton[3]);
    for (int y = 0; y <= 3; y++) {
      digitalWrite(buttonsLEDS[y+3], modeButton[y]);
    }
  }
  result = result + modeState;


  // Send data if new
  if (result != lastresult) {
    Serial.println(result);
  }
  lastresult = result;

  // wait 50ms between two reads to reduce the contact bounces
  delay(50);
}
