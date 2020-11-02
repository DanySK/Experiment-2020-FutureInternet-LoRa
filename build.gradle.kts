import org.jetbrains.kotlin.gradle.tasks.KotlinCompile
import java.util.concurrent.CompletableFuture
import java.io.File
import java.io.File.separator
import java.net.URL

plugins {
    kotlin("jvm") version "1.3.61"
}

version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib-jdk8"))
    implementation("com.google.guava:guava:28.2-jre")
    implementation("com.uchuhimo:konf:0.13.3")
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}

val configDir = File(projectDir, ".temp")
val resourcesDir = "${projectDir.absolutePath}${separator}src${separator}main${separator}resources"
val physicalNetworkDir = "$resourcesDir/config/physicalnetwork"

fun createConfigFileTask(configFile: String) = tasks.register<JavaExec>("createConfigFile") {
    dependsOn("build")

    if (configDir.exists() && configDir.isDirectory) {
        configDir.deleteRecursively()
    }
    configDir.mkdir()

    main = "it.unibo.gradle.CartesianProduct"
    args(configDir.absolutePath, configFile)
    classpath = sourceSets["main"].runtimeClasspath
}

class Job(
    private val runtime: Runtime,
    private val listOfFiles: ListOfFiles,
    private val classpathJob: String,
    private val envFile: String,
    private val outputDir: String
) : Thread() {
    val future: CompletableFuture<Int> = CompletableFuture()
    private var stop = false
    override fun run() {
        while (!stop) {
            val file = listOfFiles.getNextFile()
            if (file == null) {
                stop = true
            } else {
                runtime.exec(getCmd(file)).onExit().get()
            }
        }
        future.complete(0)
    }
    private fun getCmd(file: String) =
        "java -Xmx1700m -cp $classpathJob Simulator -cf $envFile -nf $file -of $outputDir"
}

class ListOfFiles(list: List<String>) {
    private val files = list.toMutableList()
    fun getNextFile(): String? {
        synchronized(this) {
            return if (files.isNotEmpty()) {
                files.removeAt(0)
            } else {
                null
            }
        }
    }
}

fun executeSimulations(name: String, sourceConfigFile: String, outputDir: String) = tasks.register<DefaultTask>(name) {
    val envFile = "${resourcesDir}${separator}config${separator}environment${separator}smartThermostats.xml"
    val outputDirFile = File(outputDir)
    if (!outputDirFile.exists() || !outputDirFile.isDirectory) {
        outputDirFile.mkdir()
    }

    dependsOn("build")
    dependsOn(createConfigFileTask(sourceConfigFile))
    val classpathJob = "${getSimulatorPath()}${System.getProperty("path.separator")}$resourcesDir"
    doLast {
        val runtime = Runtime.getRuntime()
        val files = ListOfFiles(configDir.listFiles().filter { it.extension == "toml" }.map { it.absolutePath })
        val jobs = (0 until runtime.availableProcessors())
            .map { Job(runtime, files, classpathJob, envFile, outputDir) }
            .map { Pair(it, it.future) }
        jobs.forEach { it.first.start() }
        jobs.forEach { it.second.get() }
    }
}

fun getSimulatorPath(): String {
    val simulatorVersion = "1.3.3"
    val simulatorUrl = "https://github.com/Placu95/DingNet/releases/download/v$simulatorVersion/DingNet-$simulatorVersion.jar"
    val simulatorFileName = "DingNet-$simulatorVersion.jar"
    val libSaveDir = "${projectDir.absolutePath}${separator}simulator"

    val folder = File(libSaveDir)
    if (!folder.exists()) {
        folder.mkdirs()
    }
    val file = File(libSaveDir, simulatorFileName)
    if (!file.exists()) {
        println("start download the simulator")
        URL(simulatorUrl).openStream().readAllBytes().also { file.appendBytes(it) }
    }

    return "$libSaveDir${separator}$simulatorFileName"
}

tasks.register<DefaultTask>("batch") {
    dependsOn(
        executeSimulations("runBatch", "$physicalNetworkDir/networkConfigs.toml",
            "${projectDir.absolutePath}${separator}data")
    )
}

tasks.register<DefaultTask>("batchRound") {
    dependsOn(
        executeSimulations("runBatch", "$physicalNetworkDir/networkConfigsRound.toml",
            "${projectDir.absolutePath}${separator}dataRound")
    )
}

tasks.register<DefaultTask>("smallBatch") {
    val outputDir = "${projectDir.absolutePath}${separator}smallBatchData"
    dependsOn(executeSimulations("runSmallBatch", "$physicalNetworkDir/fewNetworkConfigs.toml",outputDir))
    doLast {
        val numConfigs = configDir.listFiles().filter { it.extension == "toml" }.size
        val numResults = File(outputDir).listFiles().filter { it.extension == "txt" }.size
        check(numConfigs == numResults) { "the number of results' files ($numResults) is different from the number of configs' files ($numConfigs)" }
    }
}

tasks.register<JavaExec>("runWithGUI") {
    main = "Simulator"
    classpath = files(getSimulatorPath(), resourcesDir)
    args(
        "-nf", "$physicalNetworkDir/singleNetworkConfig.toml"
    )
}