
## add two more log levels between Debug = LogLevel(-1000) and Info = LogLevel(0)
## solution taken from https://discourse.julialang.org/t/creating-a-custom-log-message-categorie/29295/20

struct CustomLevel
    level::Int32
    name::String
end

const MoreInfo = CustomLevel(-100, "MoreInfo")
const DeepInfo = CustomLevel(-200, "DeepInfo")

Base.isless(a::CustomLevel, b::LogLevel) = isless(a.level, b.level)
Base.isless(a::LogLevel, b::CustomLevel) = isless(a.level, b.level)
Base.convert(::Type{LogLevel}, level::CustomLevel) = LogLevel(level.level)
Base.convert(::Type{Int32}, level::CustomLevel) = level.level

Logging.handle_message(logger::Union{ConsoleLogger, SimpleLogger,NullLogger}, level::CustomLevel, message, _module, group, id, filepath, line; kwargs...) =
    Logging.handle_message(logger, LogLevel(level), message, _module, group, id, filepath, line; kwargs...)

function set_verbosity(verbosity::Int, loggertype = ConsoleLogger)
    #logger = global_logger()
    if verbosity < 0
        global_logger(loggertype(stdout, Logging.Warn))
    elseif verbosity == 0
        global_logger(loggertype(stdout, Logging.Info))
    elseif verbosity == 1
        global_logger(loggertype(stdout, convert(LogLevel, MoreInfo.level)))
    elseif verbosity == 2
        global_logger(loggertype(stdout, convert(LogLevel, DeepInfo.level)))
    elseif verbosity >= 3
        global_logger(loggertype(stdout, Logging.Debug))
    end 
end
export MoreInfo, DeepInfo, set_verbosity