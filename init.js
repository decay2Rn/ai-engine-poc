import 'process'
db = db.getSiblingDB('entities');

db.createCollection('people_detection');
db.createCollection('face_mask_detection');

// db.people_detection.insertOne(
//  {
//     event: 'helpdev',
//     timestamp: 'EVENT_A',
//     location: 'http://rest_client_1:8080/wh'
//   }
// );
db.createUser({
    user: process.env.MONGODB_USERNAME,
    pwd: process.env.MONGODB_PASSWORD,
    roles: [{ role: "readWrite", db:  process.env.MONGODB_DATABASE}],
}); 