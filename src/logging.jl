
## add two more log levels between Debug = LogLevel(-1000) and Info = LogLevel(0)
struct GRMPLogLevel
    level::Int32
end

const DeepInfo = GRMPLogLevel(-200)
const MoreInfo = GRMPLogLevel(-100)

Base.isless(a::GRMPLogLevel, b::LogLevel) = isless(a.level, b.level)
Base.isless(a::LogLevel, b::GRMPLogLevel) = isless(a.level, b.level)
Base.convert(::Type{LogLevel}, level::GRMPLogLevel) = LogLevel(level.level)
Base.show(io::IO, level::GRMPLogLevel) =
    if level == DeepInfo
        print(io, "deepinfo")
    elseif level == MoreInfo
        print(io, "moreinfo")
    elseif level == RegInfo
        print(io, "info")
    else
        show(io, LogLevel(level))
    end

function Logging.handle_message(logger, level::GRMPLogLevel, message, _module, group, id, filepath, line)
    Logging.handle_message(logger, convert(LogLevel,level), message, _module, group, id, filepath, line)
end

Logging.disable_logging(Logging.BelowMinLevel)

function set_verbosity(verbosity::Int)
    logger = current_logger()
    if verbosity < 0
        if logger.min_level != Logging.Warn
            logger = ConsoleLogger(stdout, Logging.Warn)
            global_logger(logger)
        end
    elseif verbosity == 0
        if logger.min_level != Logging.Info
            logger = ConsoleLogger(stdout, Logging.Info)
            global_logger(logger)
        end
    elseif verbosity == 1
        if logger.min_level != MoreInfo
            logger = ConsoleLogger(stdout, MoreInfo)
            global_logger(logger)
            @logmsg MoreInfo "you can now see more info messages"
        end
    elseif verbosity == 2
        if logger.min_level != DeepInfo
            logger = ConsoleLogger(stdout, DeepInfo)
            global_logger(logger)
            @logmsg DeepInfo "you can now see deep info messages"
        end
    elseif verbosity >= 3
        if logger.min_level != Logging.Debug
            logger = ConsoleLogger(stdout, Logging.Debug)
            global_logger(logger)
            @logmsg Logging.Debug "you can now see debug messages"
        end
    end 
end
export MoreInfo, DeepInfo, set_verbosity