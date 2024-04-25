const express = require('express');
const { spawn } = require('child_process');
const bodyParser = require('body-parser');
const fs = require('fs');
const cors = require('cors'); // Import the CORS middleware
const app = express();
const port = 8080;


// Middleware to parse JSON bodies
app.use(bodyParser.json());
// Use CORS middleware
app.use(cors());

app.get('/save-tracks', (req, res) => {
  var dataToSend;
  // spawn new child process to call the python script
  const python = spawn('python', ['rs.py']);
  // collect data from script
  python.stdout.on('data', function (data) {
    console.log('Pipe data from python script ...');
    dataToSend = data.toString();
    // send data to browser inside the callback
    res.send(dataToSend);
  });
  // in close event we are sure that stream from child process is closed
  python.on('close', (code) => {
    console.log(`child process close all stdio with code ${code}`);
  });
});

app.listen(port, () => console.log(`App listening on port ${port}`));