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

byte startPin = 2;
byte pins = 52;

byte endPin = startPin + pins;
int key;
String result;
String lastresult = "";



void setup() {
  for (int x = startPin; x < endPin; x++) {
    pinMode(x, INPUT_PULLUP);
  }
  Serial.begin(115200);
}

void processInputs() {
  result = "";

  for (int pin = startPin; pin < endPin ; pin++) {

    // invert value so that closed is 1 and open is 0
    key = !digitalRead(pin);

    // construct the output string
    result = result + String(key);

  }

  // only write the result string in case the matrix has changed
  if (result != lastresult) {
    Serial.println(result);
  }
  lastresult = result;
}

void loop() {
  processInputs();

  // wait 50ms between two reads to reduce the contact bounces
  delay(50);
}
