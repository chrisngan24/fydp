//#include <I2Cdev.h>
//
//#include <helper_3dmath.h>
//#include <MPU6050.h>
#include <MPU6050_6Axis_MotionApps20.h>
//#include <MPU6050_9Axis_MotionApps41.h>


// Updates should (hopefully) always be available at https://github.com/jrowberg/i2cdevlib
// I2Cdev and MPU6050 must be installed as libraries, or else the .cpp/.h files
// for both classes must be in the include path of your project
#include "I2Cdev.h"
//#include "MPU6050.h"




// Arduino Wire library is required if I2Cdev I2CDEV_ARDUINO_WIRE implementation
// is used in I2Cdev.h
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    #include "Wire.h"
#endif

// class default I2C address is 0x68
// specific I2C addresses may be passed as a parameter here
// AD0 low = 0x68 (default for InvenSense evaluation board)
// AD0 high = 0x69
MPU6050 accelgyro;
//MPU6050 accelgyro(0x69); // <-- use for AD0 high
//ioctl("TIOCMGET"):
int16_t ax, ay, az;
uint16_t packetSize;    // expected DMP packet size (default is 42 bytes)
int16_t gx, gy, gz;
float euler[3];         // [psi, theta, phi]    Euler angle container
uint16_t fifoCount;     // count of all bytes currently in FIFO
uint8_t fifoBuffer[64]; // FIFO storage buffer
Quaternion q;           // [w, x, y, z]         quaternion container


// uncomment "OUTPUT_READABLE_ACCELGYRO" if you want to see a tab-separated
// list of the accel X/Y/Z and then gyro X/Y/Z values in decimal. Easy to read,
// not so easy to parse, and slow(er) over UART.
#define OUTPUT_READABLE_ACCELGYRO

// uncomment "OUTPUT_BINARY_ACCELGYRO" to send all 6 axes of data as 16-bit
// binary, one right after the other. This is very fast (as fast as possible
// without compression or data loss), and easy to parse, but impossible to read
// for a human.
//#define OUTPUT_BINARY_ACCELGYRO


#define LED_PIN 13
bool blinkState = false;

void setup() {
    // join I2C bus (I2Cdev library doesn't do this automatically)
    #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
        Wire.begin();
    #elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
        Fastwire::setup(400, true);
    #endif

    // initialize serial communication
    // (38400 chosen because it works as well at 8MHz as it does at 16MHz, but
    // it's really up to you depending on your project)
    Serial.begin(9600);

    // initialize device
    //Serial.println("Initializing I2C devices...");
    accelgyro.initialize();
    accelgyro.dmpInitialize();
    accelgyro.testConnection();
    accelgyro.setDMPEnabled(true);
    // verify connection
//    Serial.println("Testing device connections...");
//    Serial.println(accelgyro.testConnection() ? "MPU6050 connection successful" : "MPU6050 connection failed");
    packetSize = accelgyro.dmpGetFIFOPacketSize();


    // configure Arduino LED for
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    // read raw accel/gyro measurements from device
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

//
//    fifoCount = accelgyro.getFIFOCount();
//
//
//    while (fifoCount < packetSize) fifoCount = accelgyro.getFIFOCount();
//
//    accelgyro.getFIFOBytes(fifoBuffer, packetSize);

    fifoCount -= packetSize;
    accelgyro.dmpGetQuaternion(&q, fifoBuffer);
//    accelgyro.dmpGetEuler(euler, &q);


    #ifdef OUTPUT_READABLE_ACCELGYRO
        // display tab-separated accel/gyro x/y/z values
        // Serial.print("a/g:\t");
//        Serial.print("e1:");Serial.print(euler[0]); Serial.print(",");
//        Serial.print("e2:");Serial.print(euler[1]); Serial.print(",");
//        Serial.print("e3:");Serial.print(euler[2]); Serial.print(",");
        Serial.print("ax:");Serial.print(ax); Serial.print(",");
        Serial.print("ay:");Serial.print(ay); Serial.print(",");
        Serial.print("az:");Serial.print(az); Serial.print(",");
        Serial.print("gx:");Serial.print(gx); Serial.print(",");
        Serial.print("gy:");Serial.print(gy); Serial.print(",");
        Serial.print("gz:");Serial.println(gz);
     
    #endif

    #ifdef OUTPUT_BINARY_ACCELGYRO
        Serial.write((uint8_t)(ax >> 8)); Serial.write((uint8_t)(ax & 0xFF));
        Serial.write((uint8_t)(ay >> 8)); Serial.write((uint8_t)(ay & 0xFF));
        Serial.write((uint8_t)(az >> 8)); Serial.write((uint8_t)(az & 0xFF));
        Serial.write((uint8_t)(gx >> 8)); Serial.write((uint8_t)(gx & 0xFF));
        Serial.write((uint8_t)(gy >> 8)); Serial.write((uint8_t)(gy & 0xFF));
        Serial.write((uint8_t)(gz >> 8)); Serial.write((uint8_t)(gz & 0xFF));
    #endif

    // blink LED to indicate activity
    blinkState = !blinkState;
    digitalWrite(LED_PIN, blinkState);
    delay(100);
}
