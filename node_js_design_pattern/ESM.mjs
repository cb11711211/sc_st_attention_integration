// logger.js
// export function as log
export function log(msg) {
    console.log(`Info: ${msg}`)
}
// export const value 
export const DEFAULT_LEVEL = 'info'
// export object
export const LEVELS = {
    error: 0,
    debug: 1,
    warn: 2,
    data: 3,
    info: 4,
    verbose: 5
}
// export class
export class Logger {
    constructor (name) {
        this.name = name
    }
    log(msg) {
        console.log(`[${this.name}] ${msg}`)
    }
}
// import
import * as loggerModule from './logger.js'
console.log(loggerModule)

// import needed function
import { log } from './logger.js'
log('Hello World')
// import more than one function
import { log, Logger} from './logger.js'
log('Hello World')
const logger = new Logger('myLogger')
logger.log('Hello World')
// import as
import { log as logInfo } from './logger.js'
logInfo('Hello World')


// import default
// logger.js
export default class Logger{
    constructor(name){
        this.name = name
    }
    log(msg){
        console.log(`[${this.name}] ${msg}`)
    }
}

// main.js
import MyLogger from './logger.js'
const logger = new MyLogger('myLogger')
logger.log('Hello World')

// show default.js
import * as loggerModule from './logger.js'
console.log(loggerModule)


// mixture using default and named export
// logger.js
export default function log(msg){
    console.log(`Log: ${msg}`)
}

export function info(msg){
    log(`Info: ${msg}`)
}

// export default instance and named class
import mylog, { info } from './logger.js'

// export const HELLO = 'Γεια σου κόσμε'
// export const HELLO = 'Hello World'
// export const HELLO = 'Hola mundo'
// export const HELLO = 'Ciao mondo'
// export const HELLO = 'Witaj świecie'

const SUPPORTED_LANGUAGES = ['el', 'en', 'es', 'it', 'pl']
const selectedLanguage = process.argv[2]

if (!SUPPORTED_LANGUAGES.includes(selectedLanguage)) {
  console.error('The specified language is not supported')
  process.exit(1)
}

const translationModule = `./strings-${selectedLanguage}.js` // ①
import(translationModule) // ②
  .then((strings) => { // ③
    console.log(strings.HELLO)
  })

// read-only live binding
// counter.js
export let counter = 0
export function increment() {
    counter++
}

// main.js
import { counter, increment } from './counter.js'
console.log(counter) // 0
increment()
console.log(counter) // 1
count++ // TypeError: Assignment to constant variable


// mock-read-file.js
import fs from 'fs'
const originalReadFile = fs.readFile
let mockedResponse = null
function mockedReadFile (path, cb) {
    setImmediate(() => {
        cb(null, mockedResponse)
    })
}
export function mockEnable (response) {
    mockedResponse = response
    fs.readFile = mockedReadFile
}
export function mockDisable () {
    fs.readFile = originalReadFile
}

// main.js
import fs from 'fs'
import { mockEnable, mockDisable } from './mock-read-file.js'
mockEnable(Buffer.from('Hello World'))
fs.readFile('foo.txt', (err, data) => {
    console.log(data.toString())
    mockDisable()
    fs.readFile('foo.txt', (err, data) => {
        console.log(data.toString())
    })
})
mockDisable()