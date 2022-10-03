// define a module
const dependency = require('./dependency')

// function
function log() {
    console.log('Well done ${dependency.name}!')
}

// export API 
module.exports.run = () => {
    log()
}

// add new content to the module
exports.hello = () => {
    console.log('Hello World')
}
// or 
module.exports = () => {
    console.log('Hello World')
}

// 2.5 Using the module
// named exports
// file logger.js
exports.info = (msg) => {
    console.log(`Info: ${msg}`)
}
exports.verbose = (msg) => {
    console.log(`Verbose: ${msg}`)
}

// file main.js
const logger = require('./logger')
logger.info('This is an info message')
logger.verbose('This is a verbose message')


// exporting function
// file logger.js
module.exports = (msg) => {
    console.log(`Info: ${msg}`)
}

// alternatively
module.exports.verbose = (msg) => {
    console.log(`Verbose: ${msg}`)
}

// file main.js
const logger = require('./logger')
logger('This is an info message')
logger.verbose('This is a verbose message')


// exporting class
class Logger {
    constructor (name) {
        this.name = name
    }
    log(msg) {
        console.log(`[${this.name}] ${msg}`)
    }
    info(msg) {
        console.log(`Info: ${msg}`)
    }
    verbose(msg) {
        console.log(`Verbose: ${msg}`)
    }
}
//file main.js
const Logger = require('./logger')
const dblogger = new Logger('db')
dblogger.log('This is a log message')
const accesslogger = new Logger('access')
accesslogger.verbose('This is a verbose message')


// exporting instance
// could share the same instance across the application
// file logger.js
class Logger {
    constructor(name) {
        this.count = 0
        this.name = name
    }
    log(msg) {
        this.count++
        console.log(`[${this.name}] ${msg}`)
    }
}
module.exports = new Logger('default')

// file main.js
const logger = require('./logger')
logger.log('This is a log message')


// monkey patching
// file pather.js
require('./logger').customMsg = function() {
    console.log('This is a custom message')
}
// file main.js
require('./patcher')
const logger = require('./logger')
logger.customMsg()

